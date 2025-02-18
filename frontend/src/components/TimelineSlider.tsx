import React from 'react';

interface TimelineSliderProps {
  years: number[];
  selectedYear: number;
  onChange: (year: number) => void;
}

const TimelineSlider: React.FC<TimelineSliderProps> = ({ years, selectedYear, onChange }) => {
  return (
    <div className="mt-6 px-4">
      <div className="flex justify-between items-center">
        <span className="text-lg font-semibold">Timeline</span>
        <span className="text-gray-600">Selected Year: {selectedYear}</span>
      </div>
      
      <div className="mt-4 flex justify-between">
        {years.map((year) => (
          <button
            key={year}
            onClick={() => onChange(year)}
            className={`px-4 py-2 rounded-lg transition-all ${
              year === selectedYear
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
            }`}
          >
            {year}
          </button>
        ))}
      </div>
    </div>
  );
};

export default TimelineSlider; 